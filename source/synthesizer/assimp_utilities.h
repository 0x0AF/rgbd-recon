/* This is a utility program that helps OpenGL programmers check their program for errors.
The following functions are provided.

// Print the content of an aiScene object.
// Call this function after Assimp::Importer.ReadFile().
void printAiSceneInfo(const aiScene* scene, AiScenePrintOption option);

Written by Ying Zhu
Department of Computer Science
Georgia State University

2014

*/

#ifndef ASSIMP_UTILITIES
#define ASSIMP_UTILITIES

#include "assimp/scene.h"

#include <fstream>
#include <iostream>

enum AiScenePrintOption
{
    PRINT_AISCENE_SUMMARY,
    PRINT_AISCENE_DETAIL
};

using namespace std;

void printVector3D(string name, aiVector3D vector) { cout << name << " " << vector.x << " " << vector.y << " " << vector.z << endl; }

void printColor3D(string name, aiColor3D color) { cout << name << " " << color.r << " " << color.g << " " << color.b << endl; }

void indent(unsigned int layer)
{
    for(unsigned int m = 0; m < layer; m++)
    {
        cout << "    ";
    }
}

void printMatrix4x4(aiMatrix4x4 matrix, unsigned int layer = 0)
{
    indent(layer);
    cout << matrix.a1 << ", " << matrix.a2 << ", " << matrix.a3 << ", " << matrix.a4 << endl;
    indent(layer);
    cout << matrix.b1 << ", " << matrix.b2 << ", " << matrix.b3 << ", " << matrix.b4 << endl;
    indent(layer);
    cout << matrix.c1 << ", " << matrix.c2 << ", " << matrix.c3 << ", " << matrix.c4 << endl;
    indent(layer);
    cout << matrix.d1 << ", " << matrix.d2 << ", " << matrix.d3 << ", " << matrix.d4 << endl;
}

void printNodeTree(const aiNode *node, unsigned int layer)
{
    if(!node)
    {
        cout << "printNodeTree(): null pointer" << endl;
        return;
    }

    indent(layer);

    cout << "node: " << node->mName.C_Str();
    if(node->mNumMeshes > 0)
    {
        cout << "(Linked with mesh ";

        for(unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            cout << "#" << node->mMeshes[i] << " ";
        }
        cout << ')' << endl;
    }
    else
    {
        cout << endl;
    }

    indent(layer);
    cout << "Transformation matrix" << endl;
    printMatrix4x4(node->mTransformation, layer);

    // indent(layer);
    // cout << "Number of children " << node->mNumChildren << endl;

    cout << endl;

    for(unsigned int j = 0; j < node->mNumChildren; j++)
    {
        printNodeTree(node->mChildren[j], layer + 1);
    }
}

void printAiSceneInfo(const aiScene *scene, AiScenePrintOption option = PRINT_AISCENE_SUMMARY)
{
    if(!scene)
    {
        cout << "printAiSceneInfo(): null pointer" << endl;
        return;
    }

    cout << endl << "---------- Node Tree ----------" << endl;
    printNodeTree(scene->mRootNode, 0);

    if(scene->HasMeshes())
    {
        cout << endl << "---------- Meshes ----------" << endl;
        cout << "Total number of meshes: " << scene->mNumMeshes << endl << endl;

        for(unsigned int i = 0; i < scene->mNumMeshes; i++)
        {
            const aiMesh *currentMesh = scene->mMeshes[i];
            cout << "Mesh #" << i << endl;
            cout << "Name " << currentMesh->mName.C_Str() << endl;

            cout << "This mesh has " << currentMesh->GetNumUVChannels() << " UV(Texture) channels." << endl;
            cout << "This mesh is linked with material #" << currentMesh->mMaterialIndex << endl;

            string primitiveType = "";
            switch(currentMesh->mPrimitiveTypes)
            {
            case aiPrimitiveType_POINT:
                primitiveType = "point";
                break;
            case aiPrimitiveType_LINE:
                primitiveType = "line";
                break;
            case aiPrimitiveType_TRIANGLE:
                primitiveType = "triangle";
                break;
            case aiPrimitiveType_POLYGON:
                primitiveType = "polygon";
                break;
            default:
                primitiveType = "unknown";
                break;
            }
            cout << "Primitive type " << primitiveType << endl;

            if(currentMesh->HasPositions())
            {
                cout << "Number of vertex positions:" << currentMesh->mNumVertices << endl;

                if(option == PRINT_AISCENE_DETAIL)
                {
                    for(unsigned int j = 0; j < currentMesh->mNumVertices; j++)
                    {
                        cout << "\tvertex (" << currentMesh->mVertices[j].x << ", " << currentMesh->mVertices[j].y << ", " << currentMesh->mVertices[j].z << ")" << endl;
                    }
                }
            }
            else
            {
                cout << "There is no vertex position in mesh # " << i << endl;
            }

            if(currentMesh->HasFaces())
            {
                cout << "Number of faces:" << currentMesh->mNumFaces << endl;

                if(option == PRINT_AISCENE_DETAIL)
                {
                    for(unsigned int j = 0; j < currentMesh->mNumFaces; j++)
                    {
                        cout << "\tface #" << j << ": ";

                        for(unsigned int k = 0; k < currentMesh->mFaces[j].mNumIndices; k++)
                        {
                            cout << currentMesh->mFaces[j].mIndices[k] << ", ";
                        }
                        cout << endl;
                    }
                }
            }
            else
            {
                cout << "There is no face (element) in mesh # " << i << endl;
            }

            if(currentMesh->HasNormals())
            {
                cout << "Number of normals:" << currentMesh->mNumVertices << endl;

                if(option == PRINT_AISCENE_DETAIL)
                {
                    for(unsigned int j = 0; j < currentMesh->mNumVertices; j++)
                    {
                        cout << "\tnormal (" << currentMesh->mNormals[j].x << ", " << currentMesh->mNormals[j].y << ", " << currentMesh->mNormals[j].z << ")" << endl;
                    }
                }
            }
            else
            {
                cout << "There is no normal vectors in mesh # " << i << endl;
            }

            // Each mesh may have multiple UV(texture) channels (multi-texture). Here we only print out
            // the first channel. Call currentMesh->GetNumUVChannels() to get the number of UV channels
            // for this mesh.
            if(currentMesh->HasTextureCoords(0))
            {
                cout << "Number of texture coordinates for UV(texture) channel 0:" << currentMesh->mNumVertices << endl;

                if(option == PRINT_AISCENE_DETAIL)
                {
                    // mTextureCoords is different from mVertices or mNormals in that it is a 2D array, not a 1D array.
                    // The first dimension of this array is the texture channel for this mesh.
                    // The second dimension is the vertex index number.
                    for(unsigned int j = 0; j < currentMesh->mNumVertices; j++)
                    {
                        cout << "\ttexture coordinates (" << currentMesh->mTextureCoords[0][j].x << ", " << currentMesh->mTextureCoords[0][j].y << ")" << endl;
                    }
                }
            }
            else
            {
                cout << "There is no texture coordinate in mesh # " << i << endl;
            }

            cout << endl;
        } // end of for loop
    }     // end of if HasMeshes

    if(scene->HasMaterials())
    {
        cout << endl << "---------- Materials ----------" << endl;
        cout << "Total number of materials: " << scene->mNumMaterials << endl << endl;

        for(unsigned int i = 0; i < scene->mNumMaterials; i++)
        {
            cout << "Material #" << i << endl;

            aiMaterial *currentMaterial = scene->mMaterials[i];

            aiString matName;
            currentMaterial->Get(AI_MATKEY_NAME, matName);
            cout << "Name " << matName.C_Str() << endl;

            aiColor3D color(0.0f, 0.0f, 0.0f);
            currentMaterial->Get(AI_MATKEY_COLOR_AMBIENT, color);
            cout << "Ambient color {" << color.r << ", " << color.g << ", " << color.b << "}" << endl;

            currentMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, color);
            cout << "Diffuse color {" << color.r << ", " << color.g << ", " << color.b << "}" << endl;

            currentMaterial->Get(AI_MATKEY_COLOR_SPECULAR, color);
            cout << "Specular color {" << color.r << ", " << color.g << ", " << color.b << "}" << endl;

            float shininess = 0.0f;
            currentMaterial->Get(AI_MATKEY_SHININESS, shininess);
            cout << "Shininess " << shininess << endl;

            currentMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, color);
            cout << "Emissive color {" << color.r << ", " << color.g << ", " << color.b << "}" << endl;

            unsigned int diffuseTextureCount = currentMaterial->GetTextureCount(aiTextureType_DIFFUSE);
            cout << "Diffuse texture count " << diffuseTextureCount << endl;
            for(unsigned int k = 0; k < diffuseTextureCount; k++)
            {
                aiString textureFilePath;
                if(AI_SUCCESS == scene->mMaterials[i]->GetTexture(aiTextureType_DIFFUSE, k, &textureFilePath))
                {
                    cout << "Diffuse Texture file: " << textureFilePath.data << endl;
                }
            } // end of for diffuseTextureCount

            unsigned int specularTextureCount = currentMaterial->GetTextureCount(aiTextureType_SPECULAR);
            cout << "Specular texture count " << specularTextureCount << endl;
            for(unsigned int k = 0; k < specularTextureCount; k++)
            {
                aiString textureFilePath;
                if(AI_SUCCESS == scene->mMaterials[i]->GetTexture(aiTextureType_SPECULAR, k, &textureFilePath))
                {
                    cout << "Specular texture file: " << textureFilePath.data << endl;
                }
            } // end of for specularTextureCount

            unsigned int normalMapCount = currentMaterial->GetTextureCount(aiTextureType_NORMALS);
            cout << "Normal map count " << normalMapCount << endl;
            for(unsigned int k = 0; k < normalMapCount; k++)
            {
                aiString textureFilePath;
                if(AI_SUCCESS == scene->mMaterials[i]->GetTexture(aiTextureType_NORMALS, k, &textureFilePath))
                {
                    cout << "Specular texture file: " << textureFilePath.data << endl;
                }
            } // end of for specularTextureCount

            cout << endl;
        } // end of for mNumMaterials
    }     // end if has material

    if(scene->HasLights())
    {
        cout << endl << "---------- lights ----------" << endl;
        cout << "Total number of lights: " << scene->mNumLights << endl << endl;

        for(unsigned int i = 0; i < scene->mNumLights; i++)
        {
            aiLight *currentLight = scene->mLights[i];
            cout << "Light index: " << i << endl;
            cout << "Name: " << currentLight->mName.C_Str() << endl;
            cout << "Type: ";
            switch(currentLight->mType)
            {
            case aiLightSource_POINT:
                cout << "point light" << endl;
                break;
            case aiLightSource_DIRECTIONAL:
                cout << "directional light" << endl;
                break;
            case aiLightSource_SPOT:
                cout << "spotlight" << endl;
                break;
            default:
                cout << "unknown" << endl;
                break;
            }

            if((currentLight->mType == aiLightSource_POINT) || (currentLight->mType == aiLightSource_SPOT))
            {
                printVector3D("Position", currentLight->mPosition);
            }

            if((currentLight->mType == aiLightSource_DIRECTIONAL) || (currentLight->mType == aiLightSource_SPOT))
            {
                printVector3D("Direction", currentLight->mDirection);
            }

            printColor3D("Ambient color", currentLight->mColorAmbient);
            printColor3D("Diffuse color", currentLight->mColorDiffuse);
            printColor3D("Specular color", currentLight->mColorSpecular);

            if((currentLight->mType == aiLightSource_POINT) || (currentLight->mType == aiLightSource_SPOT))
            {
                cout << "Constant attenuation " << currentLight->mAttenuationConstant << endl;
                cout << "Linear attenuation " << currentLight->mAttenuationLinear << endl;
                cout << "Quadratic attenuation " << currentLight->mAttenuationQuadratic << endl;
            }

            if(currentLight->mType == aiLightSource_SPOT)
            {
                cout << "Inner cone angle " << currentLight->mAngleInnerCone << endl;
                cout << "Outer cone angle " << currentLight->mAngleOuterCone << endl;
            }

            cout << endl;
        } // end for
    }     // end if has lights

    if(scene->HasCameras())
    {
        cout << endl << "---------- Cameras ----------" << endl;
        cout << "Total number of cameras: " << scene->mNumCameras << endl << endl;

        for(unsigned int i = 0; i < scene->mNumCameras; i++)
        {
            aiCamera *currentCamera = scene->mCameras[i];

            cout << "Camera index: " << i << endl;
            cout << "Name: " << currentCamera->mName.C_Str() << endl;

            printVector3D("Position", currentCamera->mPosition);
            printVector3D("Look-at vector", currentCamera->mLookAt);
            printVector3D("Up vector", currentCamera->mUp);
            cout << "Aspect ratio " << currentCamera->mAspect << endl;
            cout << "Horizontal field of view " << currentCamera->mHorizontalFOV << endl;
            cout << "Near clip plane " << currentCamera->mClipPlaneNear << endl;
            cout << "Far clip plane " << currentCamera->mClipPlaneFar << endl;

            cout << "Camera matrix" << endl;
            aiMatrix4x4 currentCameraMatrix;
            currentCamera->GetCameraMatrix(currentCameraMatrix);
            printMatrix4x4(currentCameraMatrix);

            cout << endl;
        } // end fore
    }     // end if has cameras

    if(scene->HasTextures())
    {
        cout << endl << "---------- Embedded textures ----------" << endl;
        cout << "Total number of embedded textures: " << scene->mNumTextures << endl << endl;

        for(unsigned int i = 0; i < scene->mNumTextures; i++)
        {
            aiTexture *currentTexture = scene->mTextures[i];

            cout << "Texture #" << i << endl;
            cout << "Height " << currentTexture->mHeight << endl;
            cout << "Width " << currentTexture->mWidth << endl;
            // cout << "Data " << currentTexture->pcData << endl;
        }
    } // end of if HasTextures

    if(scene->HasAnimations())
    {
        cout << endl << "Has animation" << endl;
    }
}

#endif // ASSIMP_UTILITIES